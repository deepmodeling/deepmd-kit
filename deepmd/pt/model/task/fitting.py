# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from abc import (
    abstractmethod,
)
from typing import (
    Any,
    Callable,
    List,
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
from deepmd.pt.model.task.base_fitting import (
    BaseFitting,
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
from deepmd.utils.env_mat_stat import (
    StatItem,
)
from deepmd.utils.finetune import (
    get_index_between_two_maps,
    map_atom_exclude_types,
)
from deepmd.utils.path import (
    DPPath,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


class Fitting(torch.nn.Module, BaseFitting):
    # plugin moved to BaseFitting

    def __new__(cls, *args: Any, **kwargs: Any) -> "Fitting":
        if cls is Fitting:
            return BaseFitting.__new__(BaseFitting, *args, **kwargs)
        return super().__new__(cls)

    def share_params(
        self, base_class: "Fitting", shared_level: int, model_prob=1.0, protection=1e-2, resume: bool = False
    ) -> None:
        """
        Share the parameters of self to the base_class with shared_level during multitask training.
        If not start from checkpoint (resume is False),
        some separated parameters (e.g. mean and stddev) will be re-calculated across different classes.
        """
        assert self.__class__ == base_class.__class__, (
            "Only fitting nets of the same type can share params!"
        )
        if shared_level == 0:
            # only not share the bias_atom_e and the case_embd
            # link fparam buffers
            if self.numb_fparam > 0:
                if not resume:
                    base_fparam = base_class.stats["fparam"]
                    assert len(base_fparam) == self.numb_fparam
                    for ii in range(self.numb_fparam):
                        base_fparam[ii] += self.get_stats()["fparam"][ii] * model_prob
                    fparam_avg = np.array([ii.compute_avg() for ii in base_fparam])
                    fparam_std = np.array([ii.compute_std(protection=protection) for ii in base_fparam])
                    fparam_inv_std = 1.0 / fparam_std
                    base_class.fparam_avg.copy_(
                        torch.tensor(
                            fparam_avg, device=env.DEVICE, dtype=base_class.fparam_avg.dtype
                        )
                    )
                    base_class.fparam_inv_std.copy_(
                        torch.tensor(
                            fparam_inv_std, device=env.DEVICE, dtype=base_class.fparam_inv_std.dtype
                        )
                    )
                self.fparam_avg = base_class.fparam_avg
                self.fparam_inv_std = base_class.fparam_inv_std

            # link aparam buffers
            if self.numb_aparam > 0:
                if not resume:
                    base_aparam = base_class.stats["aparam"]
                    assert len(base_aparam) == self.numb_aparam
                    for ii in range(self.numb_aparam):
                        base_aparam[ii] += self.get_stats()["aparam"][ii] * model_prob
                    aparam_avg = np.array([ii.compute_avg() for ii in base_aparam])
                    aparam_std = np.array([ii.compute_std(protection=protection) for ii in base_aparam])
                    aparam_inv_std = 1.0 / aparam_std
                    base_class.aparam_avg.copy_(
                        torch.tensor(
                            aparam_avg, device=env.DEVICE, dtype=base_class.aparam_avg.dtype
                        )
                    )
                    base_class.aparam_inv_std.copy_(
                        torch.tensor(
                            aparam_inv_std, device=env.DEVICE, dtype=base_class.aparam_inv_std.dtype
                        )
                    )
                self.aparam_avg = base_class.aparam_avg
                self.aparam_inv_std = base_class.aparam_inv_std
            # the following will successfully link all the params except buffers, which need manually link.
            for item in self._modules:
                self._modules[item] = base_class._modules[item]
        else:
            raise NotImplementedError

    def save_to_file_fparam(
        self,
        stat_file_path: DPPath,
    ) -> None:
        """Save the statistics of fparam.
        Parameters
        ----------
        path : DPPath
            The path to save the statistics of fparam.
        """
        assert stat_file_path is not None
        stat_file_path.mkdir(exist_ok=True, parents=True)
        if len(self.stats) == 0:
            raise ValueError("The statistics hasn't been computed.")
        fp = stat_file_path / "fparam"
        _fparam_stat = []
        for ii in range(self.numb_fparam):
            _tmp_stat = self.stats["fparam"][ii]
            _fparam_stat.append([_tmp_stat.number, _tmp_stat.sum, _tmp_stat.squared_sum])
        _fparam_stat = np.array(_fparam_stat)
        fp.save_numpy(_fparam_stat)
        log.info(f"Save fparam stats to {fp}.")

    def save_to_file_aparam(
        self,
        stat_file_path: DPPath,
    ) -> None:
        """Save the statistics of aparam.
        Parameters
        ----------
        path : DPPath
            The path to save the statistics of aparam.
        """
        assert stat_file_path is not None
        stat_file_path.mkdir(exist_ok=True, parents=True)
        if len(self.stats) == 0:
            raise ValueError("The statistics hasn't been computed.")
        fp = stat_file_path / "aparam"
        _aparam_stat = []
        for ii in range(self.numb_aparam):
            _tmp_stat = self.stats["aparam"][ii]
            _aparam_stat.append([_tmp_stat.number, _tmp_stat.sum, _tmp_stat.squared_sum])
        _aparam_stat = np.array(_aparam_stat)
        fp.save_numpy(_aparam_stat)
        log.info(f"Save aparam stats to {fp}.")

    def restore_fparam_from_file(self, stat_file_path: DPPath) -> None:
        """Load the statistics of fparam.
        Parameters
        ----------
        path : DPPath
            The path to load the statistics of fparam.
        """
        fp = stat_file_path / "fparam"
        arr = fp.load_numpy()
        assert arr.shape == (self.numb_fparam, 3)
        _fparam_stat = []
        for ii in range(self.numb_fparam):
            _fparam_stat.append(StatItem(number=arr[ii][0], sum=arr[ii][1], squared_sum=arr[ii][2]))
        self.stats["fparam"] = _fparam_stat
        log.info(f"Load fparam stats from {fp}.")

    def restore_aparam_from_file(self, stat_file_path: DPPath) -> None:
        """Load the statistics of aparam.
        Parameters
        ----------
        path : DPPath
            The path to load the statistics of aparam.
        """
        fp = stat_file_path / "aparam"
        arr = fp.load_numpy()
        assert arr.shape == (self.numb_aparam, 3)
        _aparam_stat = []
        for ii in range(self.numb_aparam):
            _aparam_stat.append(StatItem(number=arr[ii][0], sum=arr[ii][1], squared_sum=arr[ii][2]))
        self.stats["aparam"] = _aparam_stat
        log.info(f"Load aparam stats from {fp}.")

    def compute_input_stats(
        self,
        merged: Union[Callable[[], list[dict]], list[dict]],
        protection: float = 1e-2,
        stat_file_path: Optional[DPPath] = None,
    ) -> None:
        """
        Compute the input statistics (e.g. mean and stddev) for the fittings from packed data.

        Parameters
        ----------
        merged : Union[Callable[[], list[dict]], list[dict]]
            - list[dict]: A list of data samples from various data systems.
                Each element, `merged[i]`, is a data dictionary containing `keys`: `torch.Tensor`
                originating from the `i`-th data system.
            - Callable[[], list[dict]]: A lazy function that returns data samples in the above format
                only when needed. Since the sampling process can be slow and memory-intensive,
                the lazy function helps by only sampling once.
        protection : float
            Divided-by-zero protection
        stat_file_path : Optional[DPPath]
            The path to the stat file.
        """
        if self.numb_fparam == 0 and self.numb_aparam == 0:
            # skip data statistics
            self.stats = None
            return

        self.stats = {}

        # stat fparam
        if self.numb_fparam > 0:
            if stat_file_path is not None and stat_file_path.is_dir():
                self.restore_fparam_from_file(stat_file_path)
            else:
                sampled = merged() if callable(merged) else merged
                self.stats["fparam"] = []
                cat_data = to_numpy_array(torch.cat([frame["fparam"] for frame in sampled], dim=0))
                cat_data = np.reshape(cat_data, [-1, self.numb_fparam])
                sumv = np.sum(cat_data, axis=0)
                sumv2 = np.sum(cat_data * cat_data, axis=0)
                sumn = cat_data.shape[0]
                for ii in range(self.numb_fparam):
                    self.stats["fparam"].append(
                        StatItem(
                            number=sumn,
                            sum=sumv[ii],
                            squared_sum=sumv2[ii],
                        )
                    )
                if stat_file_path is not None:
                    self.save_to_file_fparam(stat_file_path)

            fparam_avg = np.array([ii.compute_avg() for ii in self.stats["fparam"]])
            fparam_std = np.array([ii.compute_std(protection=protection) for ii in self.stats["fparam"]])
            fparam_inv_std = 1.0 / fparam_std
            log.info(f"fparam_avg is {fparam_avg}, fparam_inv_std is {fparam_inv_std}")
            self.fparam_avg.copy_(to_torch_tensor(fparam_avg))
            self.fparam_inv_std.copy_(to_torch_tensor(fparam_inv_std))

        # stat aparam
        if self.numb_aparam > 0:
            if stat_file_path is not None and stat_file_path.is_dir():
                self.restore_aparam_from_file(stat_file_path)
            else:
                sampled = merged() if callable(merged) else merged
                self.stats["aparam"] = []
                sys_sumv = []
                sys_sumv2 = []
                sys_sumn = []
                for ss_ in [frame["aparam"] for frame in sampled]:
                    ss = np.reshape(to_numpy_array(ss_), [-1, self.numb_aparam])
                    sys_sumv.append(np.sum(ss, axis=0))
                    sys_sumv2.append(np.sum(ss * ss, axis=0))
                    sys_sumn.append(ss.shape[0])
                sumv = np.sum(np.stack(sys_sumv), axis=0)
                sumv2 = np.sum(np.stack(sys_sumv2), axis=0)
                sumn = sum(sys_sumn)
                for ii in range(self.numb_aparam):
                    self.stats["aparam"].append(
                        StatItem(
                            number=sumn,
                            sum=sumv[ii],
                            squared_sum=sumv2[ii],
                        )
                    )
                if stat_file_path is not None:
                    self.save_to_file_aparam(stat_file_path)

            aparam_avg = np.array([ii.compute_avg() for ii in self.stats["aparam"]])
            aparam_std = np.array([ii.compute_std(protection=protection) for ii in self.stats["aparam"]])
            aparam_inv_std = 1.0 / aparam_std
            log.info(f"aparam_avg is {aparam_avg}, aparam_inv_std is {aparam_inv_std}")
            self.aparam_avg.copy_(to_torch_tensor(aparam_avg))
            self.aparam_inv_std.copy_(to_torch_tensor(aparam_inv_std))

    def get_stats(self) -> dict[str, List[StatItem]]:
        """Get the statistics of the fitting_net."""
        if self.stats is None:
            raise RuntimeError(
                "The statistics of fitting net has not been computed."
            )
        return self.stats


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
    neuron : list[int]
        Number of neurons in each hidden layers of the fitting net.
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
        neuron: list[int] = [128, 128, 128],
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
        self.neuron = neuron
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
        bias_atom_e = torch.tensor(
            bias_atom_e, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=device
        )
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

        if self.dim_case_embd > 0:
            self.register_buffer(
                "case_embd",
                torch.zeros(self.dim_case_embd, dtype=self.prec, device=device),
                # torch.eye(self.dim_case_embd, dtype=self.prec, device=device)[0],
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
                    np.array(self.default_fparam), dtype=self.prec, device=device
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
                    seed=child_seed(self.seed, ii),
                    trainable=trainable,
                )
                for ii in range(self.ntypes if not self.mixed_types else 1)
            ],
        )
        # set trainable
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
        model_with_new_type_stat: Optional["GeneralFitting"] = None,
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
            "@class": "Fitting",
            "@version": 4,
            "var_name": self.var_name,
            "ntypes": self.ntypes,
            "dim_descrpt": self.dim_descrpt,
            "neuron": self.neuron,
            "resnet_dt": self.resnet_dt,
            "numb_fparam": self.numb_fparam,
            "numb_aparam": self.numb_aparam,
            "dim_case_embd": self.dim_case_embd,
            "default_fparam": self.default_fparam,
            "activation_function": self.activation_function,
            "precision": self.precision,
            "mixed_types": self.mixed_types,
            "nets": self.filter_layers.serialize(),
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

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
        data = data.copy()
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

    @abstractmethod
    def _net_out_dim(self) -> int:
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
    ) -> dict[str, torch.Tensor]:
        # cast the input to internal precsion
        xx = descriptor.to(self.prec)
        nf, nloc, nd = xx.shape

        if self.numb_fparam > 0 and fparam is None:
            # use default fparam
            assert self.default_fparam_tensor is not None
            fparam = torch.tile(self.default_fparam_tensor.unsqueeze(0), [nf, 1])

        fparam = fparam.to(self.prec) if fparam is not None else None
        aparam = aparam.to(self.prec) if aparam is not None else None

        if self.remove_vaccum_contribution is not None:
            # TODO: compute the input for vaccm when remove_vaccum_contribution is set
            # Ideally, the input for vacuum should be computed;
            # we consider it as always zero for convenience.
            # Needs a compute_input_stats for vacuum passed from the
            # descriptor.
            xx_zeros = torch.zeros_like(xx)
        else:
            xx_zeros = None
        net_dim_out = self._net_out_dim()

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
            (nf, nloc, net_dim_out),
            dtype=self.prec,
            device=descriptor.device,
        )  # jit assertion
        results = {}

        if self.mixed_types:
            atom_property = self.filter_layers.networks[0](xx)
            if self.eval_return_middle_output:
                results["middle_output"] = self.filter_layers.networks[
                    0
                ].call_until_last(xx)
            if xx_zeros is not None:
                atom_property -= self.filter_layers.networks[0](xx_zeros)
            outs = (
                outs + atom_property + self.bias_atom_e[atype].to(self.prec)
            )  # Shape is [nframes, natoms[0], net_dim_out]
        else:
            if self.eval_return_middle_output:
                outs_middle = torch.zeros(
                    (nf, nloc, self.neuron[-1]),
                    dtype=self.prec,
                    device=descriptor.device,
                )  # jit assertion
                for type_i, ll in enumerate(self.filter_layers.networks):
                    mask = (atype == type_i).unsqueeze(-1)
                    mask = torch.tile(mask, (1, 1, net_dim_out))
                    middle_output_type = ll.call_until_last(xx)
                    middle_output_type = torch.where(
                        torch.tile(mask, (1, 1, self.neuron[-1])),
                        middle_output_type,
                        0.0,
                    )
                    outs_middle = outs_middle + middle_output_type
                results["middle_output"] = outs_middle
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
                atom_property = atom_property + self.bias_atom_e[type_i].to(self.prec)
                atom_property = torch.where(mask, atom_property, 0.0)
                outs = (
                    outs + atom_property
                )  # Shape is [nframes, natoms[0], net_dim_out]
        # nf x nloc
        mask = self.emask(atype).to(torch.bool)
        # nf x nloc x nod
        outs = torch.where(mask[:, :, None], outs, 0.0)
        results.update({self.var_name: outs})
        return results

    @torch.jit.export
    def get_task_dim(self) -> int:
        """Get the output dimension of the fitting net."""
        return self._net_out_dim()
