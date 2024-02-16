# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import logging
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

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
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
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
    DEVICE,
    PRECISION_DICT,
)
from deepmd.pt.utils.exclude_types import (
    AtomExcludeMask,
)
from deepmd.pt.utils.plugin import (
    Plugin,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


class Fitting(torch.nn.Module, BaseFitting):
    __plugins = Plugin()

    @staticmethod
    def register(key: str) -> Callable:
        """Register a Fitting plugin.

        Parameters
        ----------
        key : str
            the key of a Fitting

        Returns
        -------
        Fitting
            the registered Fitting

        Examples
        --------
        >>> @Fitting.register("some_fitting")
            class SomeFitting(Fitting):
                pass
        """
        return Fitting.__plugins.register(key)

    def __new__(cls, *args, **kwargs):
        if cls is Fitting:
            try:
                fitting_type = kwargs["type"]
            except KeyError:
                raise KeyError("the type of fitting should be set by `type`")
            if fitting_type in Fitting.__plugins.plugins:
                cls = Fitting.__plugins.plugins[fitting_type]
            else:
                raise RuntimeError("Unknown fitting type: " + fitting_type)
        return super().__new__(cls)

    def share_params(self, base_class, shared_level, resume=False):
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
        elif shared_level == 2:
            # share all the layers before final layer
            # the following will successfully link all the params except buffers, which need manually link.
            self._modules["filter_layers"][0].deep_layers = base_class._modules[
                "filter_layers"
            ][0].deep_layers
        elif shared_level == 3:
            # share the first layers
            # the following will successfully link all the params except buffers, which need manually link.
            self._modules["filter_layers"][0].deep_layers[0] = base_class._modules[
                "filter_layers"
            ][0].deep_layers[0]
        else:
            raise NotImplementedError

    @classmethod
    def get_stat_name(cls, ntypes, type_name="ener", **kwargs):
        """
        Get the name for the statistic file of the fitting.
        Usually use the combination of fitting net name and ntypes as the statistic file name.
        """
        if cls is not Fitting:
            raise NotImplementedError("get_stat_name is not implemented!")
        fitting_type = type_name
        return Fitting.__plugins.plugins[fitting_type].get_stat_name(
            ntypes, type_name, **kwargs
        )

    @property
    def data_stat_key(self):
        """
        Get the keys for the data statistic of the fitting.
        Return a list of statistic names needed, such as "bias_atom_e".
        """
        raise NotImplementedError("data_stat_key is not implemented!")

    def compute_or_load_stat(
        self,
        type_map: List[str],
        sampled=None,
        stat_file_path: Optional[Union[str, List[str]]] = None,
    ):
        """
        Compute or load the statistics parameters of the fitting net.
        Calculate and save the output bias to `stat_file_path`
        if `sampled` is not None, otherwise load them from `stat_file_path`.

        Parameters
        ----------
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        sampled
            The sampled data frames from different data systems.
        stat_file_path
            The path to the statistics files.
        """
        fitting_stat_key = self.data_stat_key
        if sampled is not None:
            tmp_dict = self.compute_output_stats(sampled)
            result_dict = {key: tmp_dict[key] for key in fitting_stat_key}
            result_dict["type_map"] = type_map
            self.save_stats(result_dict, stat_file_path)
        else:  # load the statistics results
            assert stat_file_path is not None, "No stat file to load!"
            result_dict = self.load_stats(type_map, stat_file_path)
        self.init_fitting_stat(**result_dict)

    def save_stats(self, result_dict, stat_file_path: str):
        """
        Save the statistics results to `stat_file_path`.

        Parameters
        ----------
        result_dict
            The dictionary of statistics results.
        stat_file_path
            The path to the statistics file(s).
        """
        log.info(f"Saving stat file to {stat_file_path}")
        np.savez_compressed(stat_file_path, **result_dict)

    def load_stats(self, type_map, stat_file_path: str):
        """
        Load the statistics results to `stat_file_path`.

        Parameters
        ----------
        type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
        stat_file_path
            The path to the statistics file(s).

        Returns
        -------
        result_dict
            The dictionary of statistics results.
        """
        fitting_stat_key = self.data_stat_key
        target_type_map = type_map
        log.info(f"Loading stat file from {stat_file_path}")
        stats = np.load(stat_file_path)
        stat_type_map = list(stats["type_map"])
        missing_type = [i for i in target_type_map if i not in stat_type_map]
        assert not missing_type, (
            f"These type are not in stat file {stat_file_path}: {missing_type}! "
            f"Please change the stat file path!"
        )
        idx_map = [stat_type_map.index(i) for i in target_type_map]
        if stats[fitting_stat_key[0]].size:  # not empty
            result_dict = {key: stats[key][idx_map] for key in fitting_stat_key}
        else:
            result_dict = {key: [] for key in fitting_stat_key}
        return result_dict

    def change_energy_bias(
        self, config, model, old_type_map, new_type_map, bias_shift="delta", ntest=10
    ):
        """Change the energy bias according to the input data and the pretrained model.

        Parameters
        ----------
        config : Dict
            The configuration.
        model : EnergyModel
            Energy model loaded pre-trained model.
        new_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        old_type_map : str
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
        finetune_data = DpLoaderSet(
            systems, ntest, config["model"], type_split=False, noise_settings=None
        )
        sampled = make_stat_input(finetune_data.systems, finetune_data.dataloaders, 1)
        # map
        sorter = np.argsort(old_type_map)
        idx_type_map = sorter[
            np.searchsorted(old_type_map, new_type_map, sorter=sorter)
        ]
        mixed_type = np.all([i.mixed_type for i in finetune_data.systems])
        numb_type = len(old_type_map)
        type_numbs, energy_ground_truth, energy_predict = [], [], []
        for test_data in sampled:
            nframes = test_data["energy"].shape[0]
            if mixed_type:
                atype = test_data["atype"].detach().cpu().numpy()
            else:
                atype = test_data["atype"][0].detach().cpu().numpy()
            assert np.array(
                [i.item() in idx_type_map for i in list(set(atype.reshape(-1)))]
            ).all(), "Some types are not in 'type_map'!"
            energy_ground_truth.append(test_data["energy"].cpu().numpy())
            if mixed_type:
                type_numbs.append(
                    np.array(
                        [(atype == i).sum(axis=-1) for i in idx_type_map],
                        dtype=np.int32,
                    ).T
                )
            else:
                type_numbs.append(
                    np.tile(
                        np.bincount(atype, minlength=numb_type)[idx_type_map],
                        (nframes, 1),
                    )
                )
            if bias_shift == "delta":
                coord = test_data["coord"].to(DEVICE)
                atype = test_data["atype"].to(DEVICE)
                box = (
                    test_data["box"].to(DEVICE)
                    if test_data["box"] is not None
                    else None
                )
                ret = model(coord, atype, box)
                energy_predict.append(
                    ret["energy"].reshape([nframes, 1]).detach().cpu().numpy()
                )
        type_numbs = np.concatenate(type_numbs)
        energy_ground_truth = np.concatenate(energy_ground_truth)
        old_bias = self.bias_atom_e[idx_type_map]
        if bias_shift == "delta":
            energy_predict = np.concatenate(energy_predict)
            bias_diff = energy_ground_truth - energy_predict
            delta_bias = np.linalg.lstsq(type_numbs, bias_diff, rcond=None)[0]
            unbias_e = energy_predict + type_numbs @ delta_bias
            atom_numbs = type_numbs.sum(-1)
            rmse_ae = np.sqrt(
                np.mean(
                    np.square(
                        (unbias_e.ravel() - energy_ground_truth.ravel()) / atom_numbs
                    )
                )
            )
            self.bias_atom_e[idx_type_map] += torch.from_numpy(
                delta_bias.reshape(-1)
            ).to(DEVICE)
            log.info(
                f"RMSE of atomic energy after linear regression is: {rmse_ae:10.5e} eV/atom."
            )
        elif bias_shift == "statistic":
            statistic_bias = np.linalg.lstsq(
                type_numbs, energy_ground_truth, rcond=None
            )[0]
            self.bias_atom_e[idx_type_map] = (
                torch.from_numpy(statistic_bias.reshape(-1))
                .type_as(self.bias_atom_e[idx_type_map])
                .to(DEVICE)
            )
        else:
            raise RuntimeError("Unknown bias_shift mode: " + bias_shift)
        log.info(
            "Change energy bias of {} from {} to {}.".format(
                str(new_type_map),
                str(old_bias.detach().cpu().numpy()),
                str(self.bias_atom_e[idx_type_map].detach().cpu().numpy()),
            )
        )
        return None


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
    distinguish_types : bool
        Neighbor list that distinguish different atomic types or not.
    rcond : float, optional
        The condition number for the regression of atomic energy.
    seed : int, optional
        Random seed.
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
        distinguish_types: bool = False,
        rcond: Optional[float] = None,
        seed: Optional[int] = None,
        exclude_types: List[int] = [],
        **kwargs,
    ):
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
        self.rcond = rcond
        self.exclude_types = exclude_types

        self.emask = AtomExcludeMask(self.ntypes, self.exclude_types)

        # init constants
        if bias_atom_e is None:
            bias_atom_e = np.zeros([self.ntypes, self.dim_out])
        bias_atom_e = torch.tensor(bias_atom_e, dtype=self.prec, device=device)
        bias_atom_e = bias_atom_e.view([self.ntypes, self.dim_out])
        if not self.use_tebd:
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
        net_dim_out = self._net_out_dim()
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
                        net_dim_out,
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

        if seed is not None:
            log.info("Set seed to %d in fitting net.", seed)
            torch.manual_seed(seed)

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
            "trainable": True,
            "atom_ener": None,
            "layer_name": None,
            "use_aparam_as_mask": False,
            "spin": None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "GeneralFitting":
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

    def get_sel_type(self) -> List[int]:
        """Get the selected atom types of this model.

        Only atoms with selected atom types have atomic contribution
        to the result of the model.
        If returning an empty list, all atom types are selected.
        """
        return []

    @abstractmethod
    def _net_out_dim(self):
        """Set the FittingNet output dim."""
        pass

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
        nf, nloc, nd = xx.shape

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

        outs = torch.zeros(
            (nf, nloc, self.dim_out),
            dtype=env.GLOBAL_PT_FLOAT_PRECISION,
            device=env.DEVICE,
        )  # jit assertion
        if self.old_impl:
            outs = torch.zeros_like(atype).unsqueeze(-1)  # jit assertion
            assert self.filter_layers_old is not None
            if self.use_tebd:
                atom_property = self.filter_layers_old[0](xx) + self.bias_atom_e[
                    atype
                ].unsqueeze(-1)
                outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
            else:
                for type_i, filter_layer in enumerate(self.filter_layers_old):
                    mask = atype == type_i
                    atom_property = filter_layer(xx)
                    atom_property = atom_property + self.bias_atom_e[type_i]
                    atom_property = atom_property * mask.unsqueeze(-1)
                    outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
        else:
            if self.use_tebd:
                atom_property = (
                    self.filter_layers.networks[0](xx) + self.bias_atom_e[atype]
                )
                outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
            else:
                net_dim_out = self._net_out_dim()
                for type_i, ll in enumerate(self.filter_layers.networks):
                    mask = (atype == type_i).unsqueeze(-1)
                    mask = torch.tile(mask, (1, 1, net_dim_out))
                    atom_property = ll(xx)
                    atom_property = atom_property + self.bias_atom_e[type_i]
                    atom_property = atom_property * mask
                    outs = outs + atom_property  # Shape is [nframes, natoms[0], 1]
        # nf x nloc
        mask = self.emask(atype)
        # nf x nloc x nod
        outs = outs * mask[:, :, None]
        return {self.var_name: outs.to(env.GLOBAL_PT_FLOAT_PRECISION)}
